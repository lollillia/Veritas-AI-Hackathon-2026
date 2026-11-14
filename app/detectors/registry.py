from importlib import import_module

from app.detectors.base import Detector

# name -> "module:Class". Imported lazily so choosing llm_vision never imports torch.
DETECTOR_PATHS: dict[str, str] = {
    "hf_pretrained": "app.detectors.hf_pretrained:HfPretrainedDetector",
    "finetuned": "app.detectors.finetuned:FinetunedDetector",
    "llm_vision": "app.detectors.llm_vision:LlmVisionDetector",
}


def build_detector(name: str, settings: dict) -> Detector:
    if name not in DETECTOR_PATHS:
        raise KeyError(f"unknown detector '{name}'")
    module_name, class_name = DETECTOR_PATHS[name].split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**settings)
