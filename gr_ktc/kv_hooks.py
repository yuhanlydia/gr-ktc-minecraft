from __future__ import annotations

from collections.abc import Sequence

import torch


class IncrementalKVRecorder:
    """Capture only newly generated K/V tokens and immediately move them to CPU."""

    def __init__(
        self,
        layer_ids: Sequence[int],
        storage_dtype: torch.dtype = torch.float16,
        pin_memory: bool = False,
    ) -> None:
        if not layer_ids:
            raise ValueError("at least one layer id is required")
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.storage_dtype = storage_dtype
        self.pin_memory = pin_memory
        self.records: dict[int, list[torch.Tensor]] = {
            layer_id: [] for layer_id in self.layer_ids
        }
        self._seen_tokens: dict[int, int] = {layer_id: 0 for layer_id in self.layer_ids}

    @torch.no_grad()
    def record_new_tokens(self, past_key_values: object) -> int:
        """Record every cache token not observed by the previous call.

        Supports legacy tuple caches and Transformers cache objects exposing
        ``to_legacy_cache``. Returns the number of newly recorded token positions.
        """
        cache = (
            past_key_values.to_legacy_cache()
            if hasattr(past_key_values, "to_legacy_cache")
            else past_key_values
        )
        counts: list[int] = []
        for layer_id in self.layer_ids:
            key, value = cache[layer_id][:2]
            if key.ndim != 4 or value.shape != key.shape:
                raise ValueError("expected K/V shape [batch, kv_heads, seq, head_dim]")
            if key.shape[0] != 1:
                raise ValueError("recorder currently supports batch size 1")

            start = self._seen_tokens[layer_id]
            end = key.shape[-2]
            if end < start:
                raise ValueError("KV cache shrank; reset recorder between generations")
            for token_index in range(start, end):
                new_k = key[0, :, token_index, :].detach().to(
                    device="cpu", dtype=self.storage_dtype, non_blocking=True
                )
                new_v = value[0, :, token_index, :].detach().to(
                    device="cpu", dtype=self.storage_dtype, non_blocking=True
                )
                state = torch.cat((new_k.flatten(), new_v.flatten()))
                if self.pin_memory and torch.cuda.is_available():
                    state = state.pin_memory()
                self.records[layer_id].append(state)
            self._seen_tokens[layer_id] = end
            counts.append(end - start)

        if len(set(counts)) != 1:
            raise ValueError(f"selected layers advanced by different amounts: {counts}")
        return counts[0]

    def stacked(self, layer_id: int) -> torch.Tensor:
        values = self.records[layer_id]
        if not values:
            return torch.empty((0, 0), dtype=self.storage_dtype)
        return torch.stack(values)

    def reset(self) -> None:
        for layer_id in self.layer_ids:
            self.records[layer_id].clear()
            self._seen_tokens[layer_id] = 0


def select_text_layers(num_hidden_layers: int, pilot: bool = True) -> list[int]:
    if num_hidden_layers < 3:
        raise ValueError("num_hidden_layers must be at least 3")
    middle = num_hidden_layers // 3
    late = (2 * num_hidden_layers) // 3
    last = num_hidden_layers - 1
    return [late, last] if pilot else [middle, late, last]

