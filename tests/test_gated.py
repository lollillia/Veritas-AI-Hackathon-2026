import io
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from PIL import Image

from app.detectors.base import Detector, DetectorError, Result
from app.detectors.gated import GATE_PROMPT, GatedDetector


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
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


class FakeInner(Detector):
    def __init__(self):
        self.calls = 0

    def predict(self, image_bytes):
        self.calls += 1
        return Result("northern_leaf_blight", 0.97, "Long lesions.",
                      scores={"northern_leaf_blight": 0.97, "gray_leaf_spot": 0.03})


def _gated(text=None, exc=None):
    inner = FakeInner()
    det = GatedDetector(model="claude-sonnet-5-5", client=FakeClient(text, exc), inner=inner)
    return det, inner


def test_not_a_corn_leaf_answers_not_corn_leaf_without_asking_the_model():
    det, inner = _gated('{"corn_leaf": false, "note": "This is a cat."}')
    r = det.predict(_img_bytes())
    assert r.label == "not_corn_leaf" and r.note == "This is a cat." and r.confidence == 1.0
    assert inner.calls == 0


def test_a_corn_leaf_is_classified_by_the_inner_model():
    det, inner = _gated('{"corn_leaf": true, "note": "A maize leaf."}')
    assert det.predict(_img_bytes()) == Result("northern_leaf_blight", 0.97, "Long lesions.")
    assert inner.calls == 1


def test_gate_sends_the_image_and_the_question_to_the_configured_model():
    det, _ = _gated('{"corn_leaf": true, "note": "x"}')
    det.predict(_img_bytes())
    kwargs = det.client.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    blocks = kwargs["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[1] == {"type": "text", "text": GATE_PROMPT}


def test_gate_reply_in_a_code_fence_is_accepted():
    det, inner = _gated('```json\n{"corn_leaf": false, "note": "A dog."}\n```')
    assert det.predict(_img_bytes()).label == "not_corn_leaf" and inner.calls == 0


@pytest.mark.parametrize("text", ["not json", '{"note": "no flag"}', '{"corn_leaf": "maybe", "note": "x"}'])
def test_unreadable_gate_reply_falls_back_to_the_inner_model(text):
    det, inner = _gated(text)
    assert det.predict(_img_bytes()).label == "northern_leaf_blight" and inner.calls == 1


def test_gate_service_failure_falls_back_to_the_inner_model():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    det, inner = _gated(exc=anthropic.APIConnectionError(request=request))
    assert det.predict(_img_bytes()).label == "northern_leaf_blight" and inner.calls == 1


def test_invalid_image_is_rejected():
    det, inner = _gated('{"corn_leaf": true, "note": "x"}')
    with pytest.raises(DetectorError, match="not a valid image"):
        det.predict(b"not an image")
    assert inner.calls == 0


def test_registry_knows_the_gated_detector():
    from app.detectors.registry import DETECTOR_PATHS
    assert DETECTOR_PATHS["gated"] == "app.detectors.gated:GatedDetector"


def test_config_for_gated_needs_the_api_key_and_the_weights(tmp_path):
    from app.config import ConfigError, load_config
    weights = tmp_path / "w"
    weights.mkdir()
    p = tmp_path / "config.yaml"
    p.write_text(f"detector: gated\ngated:\n  model: claude-sonnet-5-5\n  api_key_env: ANTHROPIC_API_KEY\n"
                 f"  weights_path: {weights}\n")
    assert load_config(str(p), env={"ANTHROPIC_API_KEY": "k"}).detector == "gated"
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_config(str(p), env={})
    p.write_text(p.read_text().replace(str(weights), str(tmp_path / "nope")))
    with pytest.raises(ConfigError, match="weights_path"):
        load_config(str(p), env={"ANTHROPIC_API_KEY": "k"})


def test_corn_leaf_result_keeps_the_inner_scores():
    det, _ = _gated('{"corn_leaf": true, "note": "A maize leaf."}')
    assert det.predict(_img_bytes()).scores == {"northern_leaf_blight": 0.97, "gray_leaf_spot": 0.03}


def test_not_corn_leaf_has_no_scores():
    det, _ = _gated('{"corn_leaf": false, "note": "A cat."}')
    assert det.predict(_img_bytes()).scores is None
