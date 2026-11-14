import pytest

from app.detectors.base import Result
from app.detectors.registry import build_detector


def test_unknown_detector_raises_keyerror():
    with pytest.raises(KeyError, match="unknown detector 'nope'"):
        build_detector("nope", {})


def test_result_rejects_unknown_label():
    with pytest.raises(ValueError):
        Result(label="blight_of_doom", confidence=0.5, note="x")


def test_result_rejects_confidence_above_one():
    with pytest.raises(ValueError):
        Result(label="healthy", confidence=1.5, note="x")
