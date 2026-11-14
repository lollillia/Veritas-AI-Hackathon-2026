from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import create_app
from app.config import load_config
from app.detectors.registry import build_detector


def build_app(config_path: str = "config.yaml") -> FastAPI:
    """Run with: uvicorn app.main:build_app --factory"""
    load_dotenv()
    config = load_config(config_path)
    detector = build_detector(config.detector, config.settings)
    return create_app(detector, config.min_confidence)
