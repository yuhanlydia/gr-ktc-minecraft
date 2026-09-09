import torch

from gr_ktc.kv_hooks import IncrementalKVRecorder


def test_memory_reset_clears_task_state():
    recorder = IncrementalKVRecorder([0])
    cache = ((torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3)),)
    recorder.record_new_tokens(cache)
    recorder.reset()
    assert recorder.stacked(0).numel() == 0
    assert recorder.record_new_tokens(cache) == 2

