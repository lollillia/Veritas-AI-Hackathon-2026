import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from app.detectors.registry import DETECTOR_PATHS


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    detector: str
    settings: dict
    min_confidence: float


def load_config(path: str = "config.yaml", env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a mapping at the top level")

    name = raw.get("detector")
    if name not in DETECTOR_PATHS:
        raise ConfigError(
            f"unknown or missing detector '{name}'; choose one of: {', '.join(DETECTOR_PATHS)}"
        )
    settings = raw.get(name) or {}
    if not isinstance(settings, dict):
        raise ConfigError(f"settings for '{name}' must be a mapping")

    min_conf = raw.get("min_confidence", 0.5)
    if not isinstance(min_conf, (int, float)) or isinstance(min_conf, bool) or not 0 <= min_conf <= 1:
        raise ConfigError("min_confidence must be a number between 0 and 1")

    if name == "llm_vision":
        var = settings.get("api_key_env", "ANTHROPIC_API_KEY")
        if not env.get(var):
            raise ConfigError(f"environment variable {var} is not set (needed by llm_vision)")
    if name == "finetuned":
        weights = settings.get("weights_path")
        if not weights or not Path(weights).exists():
            raise ConfigError(f"finetuned weights_path does not exist: {weights}")

    return Config(detector=name, settings=settings, min_confidence=float(min_conf))
