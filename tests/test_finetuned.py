import io
from types import SimpleNamespace

from PIL import Image

from app.detectors.finetuned import FinetunedDetector
from app.detectors.hf_pretrained import NOTES, HfPretrainedDetector
from app.detectors.base import Result


class FakePipeline:
    model = SimpleNamespace(config=SimpleNamespace(id2label={0: "leaf_spot"}))

    def __call__(self, image, top_k=1):
        return [{"label": "leaf_spot", "score": 0.81}]


def test_finetuned_reuses_hf_prediction_logic():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    det = FinetunedDetector(weights_path="models/corn", pipeline=FakePipeline())
    assert isinstance(det, HfPretrainedDetector)
    assert det.predict(buf.getvalue()) == Result("leaf_spot", 0.81, NOTES["leaf_spot"])
