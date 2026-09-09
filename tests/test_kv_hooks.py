import torch

from gr_ktc.kv_hooks import IncrementalKVRecorder


def test_recorder_only_appends_unseen_tokens():
    recorder = IncrementalKVRecorder([0])
    key = torch.arange(12.0).reshape(1, 1, 3, 4)
    value = key + 100
    assert recorder.record_new_tokens(((key, value),)) == 3
    assert recorder.record_new_tokens(((key, value),)) == 0
    key2 = torch.cat((key, torch.ones(1, 1, 1, 4)), dim=2)
    value2 = key2 + 100
    assert recorder.record_new_tokens(((key2, value2),)) == 1
    assert recorder.stacked(0).shape == (4, 8)

