import pytest

from gr_ktc.voyager_http import decode_observation, final_observation


def test_decodes_voyager_double_encoded_observation():
    events = decode_observation('[["observe", {"inventory": {"oak_log": 4}}]]')
    assert final_observation(events)["inventory"]["oak_log"] == 4


def test_rejects_missing_observe_event():
    with pytest.raises(ValueError, match="no observe"):
        final_observation([["onChat", {"onChat": "hello"}]])
