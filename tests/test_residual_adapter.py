import torch

from gr_ktc.residual_adapter import (
    QueryResponsiveResidualMemoryHook,
    ResidualLoRAHook,
    ScheduledResidualStateHook,
)


class TupleLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden * 2, "cache"


def test_residual_hook_adds_factorized_shift_and_cleans_up():
    layer = TupleLayer()
    hidden = torch.tensor([[[1.0, 2.0]]])
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[2.0], [3.0]])
    baseline = layer(hidden)[0]
    with ResidualLoRAHook(layer, a, b):
        adapted, cache = layer(hidden)
    assert cache == "cache"
    assert torch.allclose(adapted, baseline + torch.tensor([[[2.0, 3.0]]]))
    assert torch.equal(layer(hidden)[0], baseline)


def test_scheduled_state_hook_skips_prefill_then_advances_tokens():
    layer = TupleLayer()
    correction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with ScheduledResidualStateHook(layer, correction):
        prefill = layer(torch.zeros(1, 3, 2))[0]
        first = layer(torch.zeros(1, 1, 2))[0]
        second = layer(torch.zeros(1, 1, 2))[0]
        saturated = layer(torch.zeros(1, 1, 2))[0]
    assert torch.equal(prefill, torch.zeros_like(prefill))
    assert torch.equal(first, correction[0].view(1, 1, 2))
    assert torch.equal(second, correction[1].view(1, 1, 2))
    assert torch.equal(saturated, correction[1].view(1, 1, 2))


def test_query_responsive_memory_selects_matching_unreachable_value():
    layer = TupleLayer()
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = torch.tensor([[3.0, 4.0], [8.0, 9.0]])
    hidden = torch.tensor([[[1.0, 0.0]]])
    baseline = layer(hidden)[0]
    with QueryResponsiveResidualMemoryHook(
        layer, keys, values, temperature=0.01, top_k=1
    ):
        adapted = layer(hidden)[0]
    assert torch.allclose(adapted, baseline + values[0].view(1, 1, 2))
