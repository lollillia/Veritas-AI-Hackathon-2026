import base64
import io
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from PIL import Image

from app.detectors.base import LABELS, DetectorError, Result
from app.detectors.llm_vision import LlmVisionDetector


def _img_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 128, 0)).save(buf, format="PNG")
    return buf.getvalue()


class FakeClient:
    def __init__(self, text=None, exc=None, blocks=None):
        self.text, self.exc, self.blocks, self.kwargs = text, exc, blocks, None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        blocks = self.blocks if self.blocks is not None else [SimpleNamespace(type="text", text=self.text)]
        return SimpleNamespace(content=blocks)


def _predict(text):
    return LlmVisionDetector(client=FakeClient(text)).predict(_img_bytes())


def test_parses_plain_json_reply():
    r = _predict('{"label": "common_rust", "confidence": 0.9, "note": "Orange pustules."}')
    assert r == Result("common_rust", 0.9, "Orange pustules.")


def test_parses_fenced_json_reply():
    r = _predict('```json\n{"label": "healthy", "confidence": 0.8, "note": "Looks fine."}\n```')
    assert r.label == "healthy" and r.confidence == 0.8


def test_unknown_label_raises_detector_error():
    with pytest.raises(DetectorError, match="unreadable"):
        _predict('{"label": "blight_of_doom", "confidence": 0.5, "note": "x"}')


def test_malformed_json_raises_detector_error():
    with pytest.raises(DetectorError, match="unreadable"):
        _predict("I think it is rust, probably!")


def test_confidence_out_of_range_raises_detector_error():
    with pytest.raises(DetectorError, match="unreadable"):
        _predict('{"label": "healthy", "confidence": 7, "note": "x"}')


def test_missing_keys_raise_detector_error():
    with pytest.raises(DetectorError, match="unreadable"):
        _predict('{"label": "healthy"}')


def test_api_error_becomes_detector_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.APIConnectionError(request=req)
    with pytest.raises(DetectorError, match="unavailable"):
        LlmVisionDetector(client=FakeClient(exc=exc)).predict(_img_bytes())


def test_request_contains_jpeg_image_block_and_all_labels():
    client = FakeClient('{"label": "healthy", "confidence": 0.8, "note": "ok"}')
    LlmVisionDetector(model="claude-sonnet-5-5", client=client).predict(_img_bytes())
    kw = client.kwargs
    assert kw["model"] == "claude-sonnet-5-5"
    blocks = kw["messages"][0]["content"]
    image = next(b for b in blocks if b["type"] == "image")
    assert image["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(image["source"]["data"]).startswith(b"\xff\xd8")
    prompt = next(b for b in blocks if b["type"] == "text")["text"]
    assert all(label in prompt for label in LABELS)


def test_not_corn_leaf_reply_passes_through():
    r = _predict('{"label": "not_corn_leaf", "confidence": 0.95, "note": "This is a selfie."}')
    assert r.label == "not_corn_leaf"


def _status_error(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("nope", response=httpx.Response(status, request=req), body=None)


def test_rejected_api_key_has_its_own_message_and_is_logged(caplog):
    exc = _status_error(anthropic.AuthenticationError, 401)
    with caplog.at_level("ERROR", logger="veritas"):
        with pytest.raises(DetectorError, match="API key"):
            LlmVisionDetector(client=FakeClient(exc=exc)).predict(_img_bytes())
    assert caplog.records


def test_unknown_model_has_its_own_message():
    exc = _status_error(anthropic.NotFoundError, 404)
    with pytest.raises(DetectorError, match="model"):
        LlmVisionDetector(client=FakeClient(exc=exc)).predict(_img_bytes())


def test_client_is_built_with_timeout_and_single_retry(monkeypatch):
    captured = {}

    class Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(anthropic, "Anthropic", Spy)
    LlmVisionDetector()
    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 1


def test_reply_with_leading_thinking_block_is_parsed():
    blocks = [
        SimpleNamespace(type="thinking", thinking="Let me look at the leaf..."),
        SimpleNamespace(type="text", text='{"label": "common_rust", "confidence": 0.9, "note": "Pustules."}'),
    ]
    r = LlmVisionDetector(client=FakeClient(blocks=blocks)).predict(_img_bytes())
    assert r.label == "common_rust"


def test_reply_without_any_text_block_is_a_clear_error():
    blocks = [SimpleNamespace(type="thinking", thinking="hmm")]
    with pytest.raises(DetectorError, match="unreadable"):
        LlmVisionDetector(client=FakeClient(blocks=blocks)).predict(_img_bytes())


def test_request_leaves_room_for_thinking_plus_answer():
    # Thinking tokens count against max_tokens; 300 truncated the JSON answer mid-sentence.
    client = FakeClient('{"label": "healthy", "confidence": 0.8, "note": "ok"}')
    LlmVisionDetector(client=client).predict(_img_bytes())
    assert client.kwargs["max_tokens"] >= 2048
