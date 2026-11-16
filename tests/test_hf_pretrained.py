import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.detectors.base import LABELS, DetectorError, Result
from app.detectors.hf_pretrained import NOTES, HfPretrainedDetector


class FakePipeline:
    def __init__(self, id2label, answer=None, exc=None, answers=None):
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))
        self.answer, self.exc, self.seen_modes = answer, exc, []
        self.answers = answers if answers is not None else ([answer] if answer else [])

    def __call__(self, image, top_k=1):
        self.seen_modes.append(image.mode)
        if self.exc:
            raise self.exc
        ranked = sorted(self.answers, key=lambda a: -a["score"])
        return ranked if top_k is None else ranked[:top_k]


def _bytes(mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, (16, 16)).save(buf, format=fmt)
    return buf.getvalue()


def test_maps_model_label_to_our_label():
    pipe = FakePipeline({0: "Corn_Common_rust"}, {"label": "Corn_Common_rust", "score": 0.93})
    det = HfPretrainedDetector("m", label_map={"Corn_Common_rust": "common_rust"}, pipeline=pipe)
    r = det.predict(_bytes())
    assert r == Result("common_rust", 0.93, NOTES["common_rust"])


def test_identity_mapping_when_no_label_map():
    pipe = FakePipeline({0: "healthy"}, {"label": "healthy", "score": 0.7})
    assert HfPretrainedDetector("m", pipeline=pipe).predict(_bytes()).label == "healthy"


def test_unmapped_model_label_fails_at_construction():
    pipe = FakePipeline({0: "Corn_Common_rust", 1: "Mystery_Class"})
    with pytest.raises(ValueError, match="Mystery_Class"):
        HfPretrainedDetector("m", label_map={"Corn_Common_rust": "common_rust"}, pipeline=pipe)


def test_label_map_pointing_to_unknown_label_fails_at_construction():
    pipe = FakePipeline({0: "A"})
    with pytest.raises(ValueError, match="A"):
        HfPretrainedDetector("m", label_map={"A": "not_a_real_label"}, pipeline=pipe)


def test_rgba_input_reaches_pipeline_as_rgb():
    pipe = FakePipeline({0: "healthy"}, {"label": "healthy", "score": 0.7})
    HfPretrainedDetector("m", pipeline=pipe).predict(_bytes("RGBA"))
    assert pipe.seen_modes == ["RGB"]


def test_pipeline_exception_becomes_detector_error():
    pipe = FakePipeline({0: "healthy"}, exc=RuntimeError("boom"))
    with pytest.raises(DetectorError):
        HfPretrainedDetector("m", pipeline=pipe).predict(_bytes())


def test_note_is_nonempty_for_every_label():
    assert all(NOTES[label].strip() for label in LABELS)


def test_predict_returns_a_score_for_every_label():
    answers = [{"label": "healthy", "score": 0.2}, {"label": "common_rust", "score": 0.7}, {"label": "eyespot", "score": 0.1}]
    pipe = FakePipeline({0: "healthy", 1: "common_rust", 2: "eyespot"}, answers=answers)
    r = HfPretrainedDetector("m", pipeline=pipe).predict(_bytes())
    assert r.label == "common_rust" and r.confidence == pytest.approx(0.7)
    assert r.scores == pytest.approx({"common_rust": 0.7, "healthy": 0.2, "eyespot": 0.1})
