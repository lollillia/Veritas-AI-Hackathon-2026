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
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.kwargs = text, exc, None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


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
