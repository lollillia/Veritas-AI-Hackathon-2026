import pytest

from app.config import Config, ConfigError, load_config

LLM_YAML = """
detector: llm_vision
llm_vision:
  model: claude-sonnet-5-5
  api_key_env: ANTHROPIC_API_KEY
"""


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return str(p)


def test_valid_llm_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path, LLM_YAML), env={"ANTHROPIC_API_KEY": "k"})
    assert cfg == Config(
        detector="llm_vision",
        settings={"model": "claude-sonnet-5-5", "api_key_env": "ANTHROPIC_API_KEY"},
        min_confidence=0.5,
    )


def test_unknown_detector_names_it(tmp_path):
    with pytest.raises(ConfigError, match="mystery"):
        load_config(_write(tmp_path, "detector: mystery\n"), env={})


def test_missing_api_key_names_env_var(tmp_path):
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_config(_write(tmp_path, LLM_YAML), env={})


def test_finetuned_missing_weights_path(tmp_path):
    text = f"detector: finetuned\nfinetuned:\n  weights_path: {tmp_path}/nope\n"
    with pytest.raises(ConfigError, match="weights_path"):
        load_config(_write(tmp_path, text), env={})


def test_min_confidence_out_of_range(tmp_path):
    with pytest.raises(ConfigError, match="min_confidence"):
        load_config(_write(tmp_path, LLM_YAML + "min_confidence: 1.5\n"), env={"ANTHROPIC_API_KEY": "k"})


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "absent.yaml"), env={})


def test_min_confidence_defaults_to_half(tmp_path):
    cfg = load_config(_write(tmp_path, LLM_YAML), env={"ANTHROPIC_API_KEY": "k"})
    assert cfg.min_confidence == 0.5
