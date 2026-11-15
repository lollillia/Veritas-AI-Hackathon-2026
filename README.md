# Veritas: Corn Leaf Disease Detector

Upload a photo of a corn (maize) leaf and get one result: healthy or one of five fungal diseases (common rust, northern leaf blight, gray leaf spot, northern leaf spot, eyespot), or "not a corn leaf", with a confidence and a one-line note. Built as a hackathon demo.

The website is the same whichever model runs behind it. `config.yaml` picks one of three interchangeable detectors:

| `detector:` | What it is | Needs |
|---|---|---|
| `llm_vision` (default) | A vision LLM (Claude) via API | `ANTHROPIC_API_KEY`, internet |
| `hf_pretrained` | A pretrained corn-disease model from Hugging Face | `requirements-ml.txt`, a `model_id` and `label_map` in config |
| `finetuned` | Our own model trained with `train/train.py` | `requirements-ml.txt`, a trained `models/corn/` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # web app + LLM backend
pip install -r requirements-ml.txt         # only for hf_pretrained / finetuned / training
```

### Secrets

Never put secrets in `config.yaml`; it only holds the *name* of an environment variable.

- `ANTHROPIC_API_KEY`: key for the `llm_vision` backend.
- `HF_TOKEN`: only needed for private or gated Hugging Face models (public ones need none). `huggingface-cli login` also works.

Either `export` them in your shell (or set them in the PyCharm run configuration), or copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

## Run

```bash
uvicorn app.main:build_app --factory --reload
# open http://127.0.0.1:8000
```

If the config is wrong (unknown detector, missing key, missing weights), startup stops with a message naming the problem.

## Switching models

Edit `detector:` in `config.yaml` and restart. Low-confidence results (below `min_confidence`, default 0.5) are shown as "not sure".

### Using a Hugging Face model (`hf_pretrained`) — still to do

No model has been chosen yet; `config.yaml` has a placeholder `model_id`. To pick one:

1. Find 2–3 corn-disease image-classification models on Hugging Face and check their licenses.
2. Try each on at least 6 corn leaf photos (include one non-corn photo).
3. Set `hf_pretrained.model_id` in `config.yaml`, and map the model's own class names to ours in `label_map`, for example `Corn_(maize)___Common_rust_: common_rust`. Startup fails if any model class is left unmapped.

Note: a corn-only classifier has no "not a corn leaf" class, so it relies on the confidence threshold for that case.

### Training your own (`finetuned`)

A trained model is in `models/corn/` (gitignored). Its accuracy and limitations are in `models/corn/MODEL_PERFORMANCE.md`: about 94% on held-out photos from its training sources, but only about 38% on a dataset it never saw. To use it, set `detector: finetuned` in `config.yaml` and restart.

Data sources, licenses and label mapping are in `docs/data-sources.md`. Every image's source, label and train/dev/test split is recorded in `data/splits.csv` and `data/splits_v2.csv` (committed; the images themselves are not). Never regenerate these: they record what each model trained on; add data with `--base`.

```bash
pip install -r requirements-ml.txt
python -m data_prep.build_splits --add NAME=PATH --base data/splits_v2.csv --out data/splits_v3.csv   # add a source; existing splits stay put
python -m train.train --manifest data/splits.csv --out models/corn --epochs 5
python -m train.evaluate --manifest data/splits.csv --model models/corn --split test
```

Folder names are mapped to labels by `train/labels.py`; unknown folders (for example `Streak_virus`) are ignored. The older `--data <folder>` mode of `train.train` still works but makes its own random split.

## Adding another detector

1. Add `app/detectors/<name>.py` with a class that subclasses `Detector` and implements `predict(image_bytes) -> Result`.
2. Add one line to `DETECTOR_PATHS` in `app/detectors/registry.py`.
3. Add a section named `<name>` to `config.yaml`; its keys are passed to your constructor.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Two-minute demo script

1. "Farmers lose yield to leaf diseases they often spot too late." Open the page.
2. Drop in a diseased leaf. Show the label, confidence bar and note.
3. Drop in a photo that is not a corn leaf. Show it says so instead of guessing.
4. Open `config.yaml`, change `detector:`, restart, and run the same photo. "The website never changes; the model behind it is a plug-in."

## Resources

Related work (papers, code) and the external services the data pipeline uses are listed in [`docs/resources.md`](docs/resources.md).
