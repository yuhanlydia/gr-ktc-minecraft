import base64
from io import BytesIO

from PIL import Image

from gr_ktc.local_qwen_vl_provider import normalize_messages_for_hf


def test_normalize_messages_turns_openai_image_url_into_pil_image():
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }]

    normalized = normalize_messages_for_hf(messages)

    assert normalized[0]["content"][0] == {"type": "text", "text": "look"}
    block = normalized[0]["content"][1]
    assert block["type"] == "image"
    assert block["image"].size == (2, 2)


def test_normalize_messages_keeps_text_only_messages_unchanged_semantically():
    messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    assert normalize_messages_for_hf(messages) == messages


def test_provider_seed_reset_restarts_generation_counter_without_loading_model():
    from gr_ktc.local_qwen_vl_provider import LocalQwenVLProvider
    provider = LocalQwenVLProvider.__new__(LocalQwenVLProvider)
    provider.set_seed(42)
    assert provider._base_seed == 42
    assert provider._call_index == 0
    provider._call_index = 9
    provider.set_seed(7)
    assert provider._base_seed == 7
    assert provider._call_index == 0
