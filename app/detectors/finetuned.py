from app.detectors.hf_pretrained import HfPretrainedDetector


class FinetunedDetector(HfPretrainedDetector):
    """Our own model, saved by train/train.py in Hugging Face format.

    weights_path is a local directory; everything else is shared with HfPretrainedDetector.
    """

    def __init__(self, weights_path: str, label_map: dict[str, str] | None = None, pipeline=None):
        super().__init__(model_id=weights_path, label_map=label_map, pipeline=pipeline)
