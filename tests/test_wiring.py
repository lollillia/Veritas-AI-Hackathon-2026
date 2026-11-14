import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.config import ConfigError
from app.detectors.base import Detector, Result

LLM_YAML = "detector: llm_vision\nmin_confidence: 0.8\nllm_vision:\n  api_key_env: ANTHROPIC_API_KEY\n"


class Fake(Detector):
    def predict(self, image_bytes):
        return Result("common_rust", 0.7, "fake")


def _jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="JPEG")
    return buf.getvalue()


def test_build_app_wires_config_to_detector(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(LLM_YAML)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    built = {}

    def fake_build(name, settings):
        built["name"], built["settings"] = name, settings
        return Fake()

    monkeypatch.setattr(main, "build_detector", fake_build)
    client = TestClient(main.build_app(str(cfg)))
    body = client.post("/predict", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    assert built["name"] == "llm_vision"
    assert body["label"] == "common_rust"
    assert body["not_sure"] is True  # 0.7 < min_confidence 0.8 from the config


def test_startup_fails_clearly_on_missing_key(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(LLM_YAML)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)  # don't read a real .env
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        main.build_app(str(cfg))
